/**
 */
package nqc.provider;


import java.util.Collection;
import java.util.List;

import nqc.NqcFactory;
import nqc.NqcPackage;
import nqc.Subroutine;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.ResourceLocator;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ItemProviderAdapter;
import org.eclipse.emf.edit.provider.ViewerNotification;

/**
 * This is the item provider adapter for a {@link nqc.Subroutine} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class SubroutineItemProvider
	extends ItemProviderAdapter
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
	public SubroutineItemProvider(AdapterFactory adapterFactory) {
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

			addNamePropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Name feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addNamePropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_Subroutine_Name_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_Subroutine_Name_feature", "_UI_Subroutine_type"),
				 NqcPackage.eINSTANCE.getSubroutine_Name(),
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
			childrenFeatures.add(NqcPackage.eINSTANCE.getSubroutine_Statements());
			childrenFeatures.add(NqcPackage.eINSTANCE.getSubroutine_LocalVariables());
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
	 * This returns Subroutine.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/Subroutine"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((Subroutine)object).getName();
		return label == null || label.length() == 0 ?
			getString("_UI_Subroutine_type") :
			getString("_UI_Subroutine_type") + " " + label;
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

		switch (notification.getFeatureID(Subroutine.class)) {
			case NqcPackage.SUBROUTINE__NAME:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case NqcPackage.SUBROUTINE__STATEMENTS:
			case NqcPackage.SUBROUTINE__LOCAL_VARIABLES:
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
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createAcquireStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createAssignmentStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBlockStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createBreakStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createContinueStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createDoWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createEmptyStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createForStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createGoToStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createIfStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createMonitorStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createQuaternaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createRepeatStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createReturnStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSenaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createStartStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createStopStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSubroutineCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createSwitchStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createTernaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createUntilStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_Statements(),
				 NqcFactory.eINSTANCE.createWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getSubroutine_LocalVariables(),
				 NqcFactory.eINSTANCE.createVariable()));
	}

	/**
	 * Return the resource locator for this item provider's resources.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public ResourceLocator getResourceLocator() {
		return NqcEditPlugin.INSTANCE;
	}

}
