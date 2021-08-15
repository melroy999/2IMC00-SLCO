/**
 */
package poosl.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ViewerNotification;

import poosl.LoopExpression;
import poosl.PooslFactory;
import poosl.PooslPackage;

/**
 * This is the item provider adapter for a {@link poosl.LoopExpression} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class LoopExpressionItemProvider
	extends ExpressionItemProvider
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
	public LoopExpressionItemProvider(AdapterFactory adapterFactory) {
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

		}
		return itemPropertyDescriptors;
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
			childrenFeatures.add(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION);
			childrenFeatures.add(PooslPackage.Literals.LOOP_EXPRESSION__BODY);
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
	 * This returns LoopExpression.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/LoopExpression"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_LoopExpression_type");
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

		switch (notification.getFeatureID(LoopExpression.class)) {
			case PooslPackage.LOOP_EXPRESSION__CONDITION:
			case PooslPackage.LOOP_EXPRESSION__BODY:
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
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createAssignment()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createConditionalExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createConstantExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createCurrentTime()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createPrimitiveDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createDataObjectCreation()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createLoopExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createParameterExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createReferenceSelf()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createReturnExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__CONDITION,
				 PooslFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createAssignment()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createConditionalExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createConstantExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createCurrentTime()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createPrimitiveDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createDataObjectCreation()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createLoopExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createParameterExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createReferenceSelf()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createReturnExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.LOOP_EXPRESSION__BODY,
				 PooslFactory.eINSTANCE.createVariableExpression()));
	}

	/**
	 * This returns the label text for {@link org.eclipse.emf.edit.command.CreateChildCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getCreateChildText(Object owner, Object feature, Object child, Collection<?> selection) {
		Object childFeature = feature;
		Object childObject = child;

		boolean qualify =
			childFeature == PooslPackage.Literals.LOOP_EXPRESSION__CONDITION ||
			childFeature == PooslPackage.Literals.LOOP_EXPRESSION__BODY;

		if (qualify) {
			return getString
				("_UI_CreateChild_text2",
				 new Object[] { getTypeText(childObject), getFeatureText(childFeature), getTypeText(owner) });
		}
		return super.getCreateChildText(owner, feature, child, selection);
	}

}
