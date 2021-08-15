/**
 */
package poosl.provider;


import java.util.Collection;
import java.util.List;

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
import org.eclipse.emf.edit.provider.ViewerNotification;

import poosl.MessageSend;
import poosl.PooslFactory;
import poosl.PooslPackage;

/**
 * This is the item provider adapter for a {@link poosl.MessageSend} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class MessageSendItemProvider
	extends StatementItemProvider
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
	public MessageSendItemProvider(AdapterFactory adapterFactory) {
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

			addPortPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Port feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addPortPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_MessageSend_port_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_MessageSend_port_feature", "_UI_MessageSend_type"),
				 PooslPackage.Literals.MESSAGE_SEND__PORT,
				 true,
				 false,
				 true,
				 null,
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
			childrenFeatures.add(PooslPackage.Literals.MESSAGE_SEND__MESSAGE);
			childrenFeatures.add(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS);
			childrenFeatures.add(PooslPackage.Literals.MESSAGE_SEND__EXPRESSION_LISTS);
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
	 * This returns MessageSend.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/MessageSend"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_MessageSend_type");
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

		switch (notification.getFeatureID(MessageSend.class)) {
			case PooslPackage.MESSAGE_SEND__MESSAGE:
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
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
				(PooslPackage.Literals.MESSAGE_SEND__MESSAGE,
				 PooslFactory.eINSTANCE.createOutgoingMessage()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createAssignment()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createConditionalExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createConstantExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createCurrentTime()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createPrimitiveDataMethodCall()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createDataObjectCreation()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createLoopExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createParameterExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createReferenceSelf()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createReturnExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSIONS,
				 PooslFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.MESSAGE_SEND__EXPRESSION_LISTS,
				 PooslFactory.eINSTANCE.createExpressionList()));
	}

}
