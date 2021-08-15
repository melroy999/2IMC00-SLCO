/**
 */
package nqc.provider;


import java.util.Collection;
import java.util.List;

import nqc.NqcFactory;
import nqc.NqcPackage;
import nqc.UnaryExpression;
import nqc.UnaryOperatorEnum;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ViewerNotification;

/**
 * This is the item provider adapter for a {@link nqc.UnaryExpression} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class UnaryExpressionItemProvider
	extends CompoundExpressionItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryExpressionItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addOperatorPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Operator feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addOperatorPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_UnaryExpression_Operator_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_UnaryExpression_Operator_feature", "_UI_UnaryExpression_type"),
				 NqcPackage.eINSTANCE.getUnaryExpression_Operator(),
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(NqcPackage.eINSTANCE.getUnaryExpression_Operand());
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns UnaryExpression.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/UnaryExpression"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		UnaryOperatorEnum labelValue = ((UnaryExpression)object).getOperator();
		String label = labelValue == null ? null : labelValue.toString();
		return label == null || label.length() == 0 ?
			getString("_UI_UnaryExpression_type") :
			getString("_UI_UnaryExpression_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(UnaryExpression.class)) {
			case NqcPackage.UNARY_EXPRESSION__OPERATOR:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case NqcPackage.UNARY_EXPRESSION__OPERAND:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryExpression_Operand(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));
	}

}
